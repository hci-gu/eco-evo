# import lib.constants as const
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from lib.config import const
from lib.config.settings import Settings
from lib.config.species import build_species_map
from lib.model import Model
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations, plot_biomass
from lib.data_manager import update_generations_data, process_data
from lib.behaviour import run_all_scenarios
from lib.evaluate import predict_years
import numpy as np
import random
import copy
import time
from lib.environments.petting_zoo import env
import optuna

# ---- Parallel worker plumbing ----
_WORKER_RUNNER = None  # one runner per proces

def noop(a, b, c):
    pass

class PettingZooRunner():
    def __init__(self, settings: Settings, render_mode="none"):
        # Create the environment (render_mode can be "none" if visualization is not needed)
        self.species_map = build_species_map(settings)
        self.env = env(settings=settings, species_map=self.species_map, render_mode=render_mode)
        self.settings = settings
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        self.current_generation = 0

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.ACTING_SPECIES]
        # Create a population for each species.
        self.population = {
            species: [Model() for _ in range(self.settings.num_agents)]
            for species in self.species_list
        }
        # Track best fitness and best model for each species.
        self.best_fitness = {species: -float('inf') for species in self.species_list}
        self.best_agent = {species: None for species in self.species_list}
        self.agent_index = 0
        self.eval_index = 0

    def run(self, candidates, species_being_evaluated = "cod", seed = None, is_evaluation = False, callback = noop):
        self.env.reset(seed)
        steps = 0
        episode_length = 0
        fitness = 0
        callback(self.env.world, fitness, False)

        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()) and episode_length < self.settings.max_steps:
            steps += 1
            agent = self.env.agent_selection
            if agent == "plankton":
                self.env.step(self.empty_action)
            else:
                obs, reward, termination, truncation, info = self.env.last()
                candidate = candidates[agent]
                action_values = candidate.forward(obs.reshape(-1, 135))
                action_values = action_values.reshape(self.settings.world_size, self.settings.world_size, 5)
                # mean_action_values = action_values.mean(axis=(0, 1))

                self.env.step(action_values)
            
            if steps % 4 == 0:
                episode_length += 1
                species_biomass = self.env.get_fitness(species_being_evaluated)
                fitness += species_biomass
                if (is_evaluation == False or self.env.render_mode != "none"):
                    process_data({
                        'species': species_being_evaluated if not is_evaluation else None,
                        'agent_index': self.agent_index,
                        'eval_index': self.eval_index,
                        'step': episode_length,
                        'world': self.env.world
                    }, self.env.plot_data)
                callback(self.env.world, fitness, False)
        
        print("end of simulation", fitness, episode_length)
        callback(self.env.world, fitness, True)
        return fitness, episode_length, self.env.reason

    def evaluate_population(self):
        """
        Evaluate each model in the population for every species.
        When evaluating a candidate for a species S, for every other species (not S),
        we use the best model from previous generations (if available) to decide their actions.
        Returns a dictionary mapping species to a list of (chromosome, fitness) tuples.
        """
        eval_seeds = [int(random.random() * 100000) for _ in range(self.settings.agent_evaluations)]
        fitnesses = {species: [] for species in self.species_list}

        end_reasons = []
        for idx in range(self.settings.num_agents):
            self.agent_index = idx
            for species in self.species_list:
                evaluation_candidate = self.population[species][idx]
                other_species = copy.deepcopy(self.species_list)
                other_species.remove(species)

                evals_fitness = []
                eval_species = {
                    species: evaluation_candidate
                }

                for eval_index in range(self.settings.agent_evaluations):
                    start_time = time.time()
                    self.eval_index = eval_index
                    # pick random agent from other species
                    for other_species_name in other_species:
                        other_species_idx = random.randint(0, self.settings.num_agents - 1)
                        other_species_candidate = self.population[other_species_name][other_species_idx]
                        eval_species[other_species_name] = other_species_candidate
                    
                    fitness, episode_length, end_reason = self.run(eval_species, species, eval_seeds[eval_index])
                    end_reasons.append(end_reason)
                    while (episode_length == 0):
                        print("something went wrong update this seed")
                        eval_seeds[eval_index] = int(random.random() * 100000)
                        fitness, episode_length = self.run(eval_species, species, eval_seeds[eval_index])
                    
                    evals_fitness.append(fitness)
                    end_time = time.time()
                    print(f'idx {idx}, eval {eval_index}, fitness {fitness:.1f}, episode_length {episode_length}, steps/sec {episode_length/(end_time - start_time):.2f}')
                
                avg_fitness = sum(evals_fitness) / len(evals_fitness)
                behaviour_score = run_all_scenarios(self.settings, self.species_map, evaluation_candidate, species, False)
                avg_fitness = avg_fitness * behaviour_score

                fitnesses[species].append((evaluation_candidate.state_dict(), avg_fitness))
                print(f'finished eval for species: {species}, fitness: {avg_fitness:.1f}')

        print("End reasons:", {reason: end_reasons.count(reason) for reason in set(end_reasons)})
        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = {}
        fittest_agents_for_generation = {}
        for species in self.species_list:
            fittest_agent = max(fitnesses[species], key=lambda x: x[1])
            fittest_agents_for_generation[species] = Model(chromosome=fittest_agent[0])

        for species in self.species_list:
            current_population = fitnesses[species]

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
            if best_for_species[1] > self.best_fitness[species] + 0.01:
                self.best_fitness[species] = best_for_species[1]
                self.best_agent[species] = best_for_species[0]
                model = Model(chromosome=self.best_agent[species])
                model.save(f'{self.settings.folder}/agents/{self.current_generation}_${species}_{self.best_fitness[species]}.npy')

            # add best agent to the population
            next_pop.append(self.best_agent[species])

            new_population[species] = [Model(chromosome=chrom) for chrom in next_pop]
        
            if (self.current_generation < 25):
                new_population[species].append(Model())
                new_population[species].append(Model())
                
        for species in self.species_list:
            # shuffle the population
            random.shuffle(new_population[species])

        self.population = new_population

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        print(f"Generation {self.current_generation} complete. Fitnesses: { {sp: max(fit, key=lambda x: x[1])[1] for sp, fit in fitnesses.items()} }")
        generations_data = update_generations_data(self.settings, self.current_generation, self.env.plot_data)
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
