import lib.constants as const
from lib.model import SingleSpeciesModel
import lib.evolution as evolution  # Your NumPy-based evolution functions
from lib.visualize import plot_generations, plot_biomass
from lib.data_manager import data_loop, update_generations_data, process_data
import numpy as np
import random
import copy
from lib.environments.petting_zoo import env

def noop(a, b):
    pass

class PettingZooRunner():
    def __init__(self, render_mode="none"):
        # Create the environment (render_mode can be "none" if visualization is not needed)
        self.env = env(render_mode=render_mode)
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        self.current_generation = 0

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.SPECIES_MAP.keys() if species != "plankton"]
        # Create a population for each species.
        self.population = {
            species: [SingleSpeciesModel() for _ in range(const.NUM_AGENTS)]
            for species in self.species_list
        }
        # Track best fitness and best model for each species.
        self.best_fitness = {species: -float('inf') for species in self.species_list}
        self.best_agent = {species: None for species in self.species_list}
        self.agent_index = 0
        self.eval_index = 0

    def run(self, candidates, species_being_evaluated = "cod", is_evaluation = False, callback = noop):
        self.env.reset()
        steps = 0
        episode_length = 0
        fitness = 0
        callback(self.env.world, fitness)

        while not all(self.env.terminations.values()) and not all(self.env.truncations.values()) and episode_length < const.MAX_STEPS:
            steps += 1
            agent = self.env.agent_selection
            if agent == "plankton":
                self.env.step(self.empty_action)
            else:
                obs, reward, termination, truncation, info = self.env.last()
                candidate = candidates[agent]
                action_values = candidate.forward(obs.reshape(-1, 99))
                action_values = action_values.reshape(const.WORLD_SIZE, const.WORLD_SIZE, const.AVAILABLE_ACTIONS)

                self.env.step(action_values)
            
            if steps % 4 == 0:
                episode_length += 1
                species_biomass = self.env.get_fitness(species_being_evaluated)
                fitness += species_biomass
                agents_data = process_data({
                    'species': species_being_evaluated if not is_evaluation else None,
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
        fitnesses = {species: [] for species in self.species_list}

        for idx in range(const.NUM_AGENTS):
            self.agent_index = idx
            for species in self.species_list:
                evaluation_candidate = self.population[species][idx]
                other_species = copy.deepcopy(self.species_list)
                other_species.remove(species)

                evals_fitness = []
                eval_species = {
                    species: evaluation_candidate
                }

                for eval_index in range(const.AGENT_EVALUATIONS):
                    self.eval_index = eval_index
                    # pick random agent from other species
                    for other_species_name in other_species:
                        other_species_idx = random.randint(0, const.NUM_AGENTS - 1)
                        other_species_candidate = self.population[other_species_name][other_species_idx]
                        eval_species[other_species_name] = other_species_candidate
                    
                    fitness, episode_length = self.run(eval_species, species)
                    evals_fitness.append(fitness)
                    print("idx", idx, "eval", eval_index, "fitness", fitness, "episode_length", episode_length)
                
                avg_fitness = sum(evals_fitness) / len(evals_fitness)
                fitnesses[species].append((evaluation_candidate.state_dict(), avg_fitness))
                print("finished eval for species", species, ", fitness", avg_fitness)
        return fitnesses

    def evolve_population(self, fitnesses):
        """
        Given a dictionary of fitnesses (per species), evolve each species' population.
        """
        new_population = {}
        fittest_agents_for_generation = {}
        for species in self.species_list:
            fittest_agent = max(fitnesses[species], key=lambda x: x[1])
            fittest_agents_for_generation[species] = SingleSpeciesModel(chromosome=fittest_agent[0])
        
        # # run an episode length evaluation for each species and see if they should replace the best agent
        # for species in self.species_list:
        #     # print("EVALUATE")
        #     # print(fittest_agents_for_generation)
        #     _, episode_length = self.run(fittest_agents_for_generation, species)
        #     if (episode_length > self.best_fitness[species]):
        #         self.best_fitness[species] = episode_length
        #         self.best_agent[species] = fittest_agents_for_generation[species]
        #         model = self.best_agent[species]
        #         model.save(f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_${species}_{self.best_fitness[species]}.npy')
        #         # TODO compare with all time best not just generation best when doing evaluation like this

        for species in self.species_list:
            current_population = fitnesses[species]

            # Select elites.
            elites = evolution.elitism_selection(current_population, const.ELITISM_SELECTION)
            next_pop = []

            while len(next_pop) < const.NUM_AGENTS:
                (p1, _), (p2, _) = evolution.tournament_selection(elites, 2, const.TOURNAMENT_SELECTION)
                c1_weights, c2_weights = evolution.crossover(p1, p2)
                current_mutation_rate = max(const.MIN_MUTATION_RATE,
                                            const.INITIAL_MUTATION_RATE * (const.MUTATION_RATE_DECAY ** self.current_generation))
                evolution.mutation(c1_weights, current_mutation_rate, current_mutation_rate)
                evolution.mutation(c2_weights, current_mutation_rate, current_mutation_rate)
                next_pop.append(c1_weights)
                next_pop.append(c2_weights)

            # Update best fitness/agent.
            best_for_species = max(current_population, key=lambda x: x[1])
            if best_for_species[1] > self.best_fitness[species]:
                self.best_fitness[species] = best_for_species[1]
                self.best_agent[species] = best_for_species[0]
                model = SingleSpeciesModel(chromosome=fittest_agent[0])
                model.save(f'{const.CURRENT_FOLDER}/agents/{self.current_generation}_${species}_{self.best_fitness[species]}.npy')

            # add best agent to the population
            next_pop.append(self.best_agent[species])
            new_population[species] = [SingleSpeciesModel(chromosome=chrom) for chrom in next_pop]
        
        for species in self.species_list:
            # shuffle the population
            random.shuffle(new_population[species])
        
    
        self.population = new_population

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        print(f"Generation {self.current_generation} complete. Fitnesses: { {sp: max(fit, key=lambda x: x[1])[1] for sp, fit in fitnesses.items()} }")
        print(self.env.plot_data.keys())
        generations_data = update_generations_data(self.current_generation, self.env.plot_data)
        plot_generations(generations_data)
        self.env.plot_data = {}
        # Evolve the population based on fitnesses.
        self.evolve_population(fitnesses)

    def train(self, generations=const.GENERATIONS_PER_RUN):
        for _ in range(generations):
            self.run_generation()
            # Optionally, save best models or log additional statistics.

    def evaluate(self, model_paths = [], callback = noop):
        candidates = {}
        for model_path in model_paths:
            model = SingleSpeciesModel(
                chromosome=np.load(model_path['path'])
            )
            candidates[model_path['species']] = model

        return self.run(candidates, "cod", True, callback)
