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
_WORKER_POPULATION_STATE = None
_WORKER_MODEL_CACHE = None

def noop(a, b, c):
    pass

class PettingZooRunner():
    def __init__(
        self,
        settings: Settings,
        render_mode: str = "none",
        map_folder: str = "maps/baltic",
        build_population: bool = True,
    ):
        # Create the environment (render_mode can be "none" if visualization is not needed)
        self.species_map = build_species_map(settings)
        self.env = env(
            settings=settings,
            species_map=self.species_map,
            render_mode=render_mode,
            map_folder=map_folder,
        )
        self.settings = settings
        self.empty_action = self.env.action_space("plankton").sample()
        self.env.reset()
        self.current_generation = 0
        self.rng = random.Random(None if self.settings.seed < 0 else self.settings.seed)

        # Use all species in the simulation (including plankton, if desired).
        self.species_list = [species for species in const.ACTING_SPECIES]
        # Create a population for each species (training only).
        self.population = {}
        if build_population:
            self.population = {
                species: [Model() for _ in range(self.settings.num_agents)]
                for species in self.species_list
            }
        # Track best fitness and best model for each species.
        self.best_fitness = {species: -float('inf') for species in self.species_list}
        self.best_agent = {species: None for species in self.species_list}
        self.agent_index = 0
        self.eval_index = 0

    def run(self, candidates, species_being_evaluated = "cod", seed = None, is_evaluation = False, callback = noop, collect_plot_data = True, python_random_seed = None):
        if python_random_seed is not None:
            random.seed(python_random_seed)
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
                if collect_plot_data and self.settings.enable_plotting and (is_evaluation == False or self.env.render_mode != "none"):
                    process_data({
                        'species': species_being_evaluated if not is_evaluation else None,
                        'agent_index': self.agent_index,
                        'eval_index': self.eval_index,
                        'step': episode_length,
                        'world': self.env.world
                    }, self.env.plot_data)
                callback(self.env.world, fitness, False)
        
        if self.settings.enable_logging:
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
        if self.settings.seed >= 0:
            # Deterministic per-generation RNG for eval setup.
            eval_rng = random.Random(self.settings.seed + self.current_generation * 1_000_003)
        else:
            eval_rng = self.rng

        # Same eval seeds for every agent in the generation.
        eval_seeds = [eval_rng.randrange(100000) for _ in range(self.settings.agent_evaluations)]
        fitnesses = {species: [] for species in self.species_list}

        end_reasons = []
        tasks = []
        population_state = {
            species: [m.state_dict() for m in self.population[species]]
            for species in self.species_list
        }
        for idx in range(self.settings.num_agents):
            for species in self.species_list:
                other_species = [s for s in self.species_list if s != species]
                for eval_index in range(self.settings.agent_evaluations):
                    other_indices = {}
                    # pick random agent from other species
                    for other_species_name in other_species:
                        other_species_idx = eval_rng.randint(0, self.settings.num_agents - 1)
                        other_indices[other_species_name] = other_species_idx

                    tasks.append({
                        "agent_index": idx,
                        "species": species,
                        "eval_index": eval_index,
                        "other_indices": other_indices,
                        "map_seed": eval_seeds[eval_index],
                        "python_random_seed": eval_rng.randrange(2**32),
                    })

        local_model_cache = {species: {} for species in self.species_list}

        def _get_local_model(species, idx):
            cache = local_model_cache[species]
            model = cache.get(idx)
            if model is None:
                model = Model(chromosome=population_state[species][idx])
                cache[idx] = model
            return model

        def _run_eval_task_local(task):
            self_model = _get_local_model(task["species"], task["agent_index"])
            eval_species = {task["species"]: self_model}
            for sp, idx in task["other_indices"].items():
                eval_species[sp] = _get_local_model(sp, idx)

            self.agent_index = task["agent_index"]
            self.eval_index = task["eval_index"]

            map_seed = task["map_seed"]
            python_random_seed = task["python_random_seed"]
            start_time = time.time()
            fitness, episode_length, end_reason = self.run(
                eval_species,
                task["species"],
                map_seed,
                is_evaluation=False,
                callback=noop,
                collect_plot_data=True,
                python_random_seed=python_random_seed,
            )
            while episode_length == 0:
                if self.settings.enable_logging:
                    print("something went wrong update this seed")
                map_seed = random.randrange(100000)
                python_random_seed = random.randrange(2**32)
                fitness, episode_length, end_reason = self.run(
                    eval_species,
                    task["species"],
                    map_seed,
                    is_evaluation=False,
                    callback=noop,
                    collect_plot_data=True,
                    python_random_seed=python_random_seed,
                )

            end_time = time.time()
            return {
                "agent_index": task["agent_index"],
                "species": task["species"],
                "eval_index": task["eval_index"],
                "fitness": fitness,
                "episode_length": episode_length,
                "end_reason": end_reason,
                "steps_per_sec": episode_length / max(end_time - start_time, 1e-9),
            }

        results = []
        if self.settings.num_workers <= 1:
            for task in tasks:
                results.append(_run_eval_task_local(task))
        else:
            with ProcessPoolExecutor(
                max_workers=self.settings.num_workers,
                initializer=_init_worker,
                initargs=(self.settings, self.env.render_mode, self.env.map_folder, population_state),
            ) as executor:
                # Small tasks improve load-balancing across workers.
                for res in executor.map(_run_eval_task_worker, tasks, chunksize=1):
                    results.append(res)

        # Aggregate results in deterministic order.
        results_by_key = {(r["agent_index"], r["species"], r["eval_index"]): r for r in results}
        for idx in range(self.settings.num_agents):
            self.agent_index = idx
            for species in self.species_list:
                evals_fitness = []
                for eval_index in range(self.settings.agent_evaluations):
                    r = results_by_key[(idx, species, eval_index)]
                    self.eval_index = eval_index
                    end_reasons.append(r["end_reason"])
                    evals_fitness.append(r["fitness"])
                    if self.settings.enable_logging:
                        print(f'idx {idx}, eval {eval_index}, fitness {r["fitness"]:.1f}, episode_length {r["episode_length"]}, steps/sec {r["steps_per_sec"]:.2f}')

                    # Minimal plot data update for parallel runs (preserves generation stats).
                    if self.settings.num_workers > 1 and self.settings.enable_plotting:
                        process_data({
                            'species': species,
                            'agent_index': idx,
                            'eval_index': eval_index,
                            'step': r["episode_length"],
                        }, self.env.plot_data)

                avg_fitness = sum(evals_fitness) / len(evals_fitness)
                fitnesses[species].append((self.population[species][idx].state_dict(), avg_fitness))
                if self.settings.enable_logging:
                    print(f'finished eval for species: {species}, fitness: {avg_fitness:.1f}')

        if self.settings.enable_logging:
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
            self.rng.shuffle(new_population[species])

        self.population = new_population

    def run_generation(self):
        # Evaluate the current population.
        fitnesses = self.evaluate_population()
        self.current_generation += 1
        if self.settings.enable_logging:
            print(f"Generation {self.current_generation} complete. Fitnesses: { {sp: max(fit, key=lambda x: x[1])[1] for sp, fit in fitnesses.items()} }")
        if self.settings.enable_plotting:
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


def _init_worker(settings: Settings, render_mode: str, map_folder: str, population_state):
    global _WORKER_RUNNER
    global _WORKER_POPULATION_STATE
    global _WORKER_MODEL_CACHE
    _WORKER_RUNNER = PettingZooRunner(
        settings=settings,
        render_mode=render_mode,
        map_folder=map_folder,
        build_population=False,
    )
    _WORKER_POPULATION_STATE = population_state
    _WORKER_MODEL_CACHE = {species: {} for species in population_state}


def _run_eval_task_worker(task):
    global _WORKER_RUNNER
    global _WORKER_POPULATION_STATE
    global _WORKER_MODEL_CACHE
    runner = _WORKER_RUNNER

    def _get_worker_model(species, idx):
        cache = _WORKER_MODEL_CACHE[species]
        model = cache.get(idx)
        if model is None:
            model = Model(chromosome=_WORKER_POPULATION_STATE[species][idx])
            cache[idx] = model
        return model

    self_model = _get_worker_model(task["species"], task["agent_index"])
    eval_species = {task["species"]: self_model}
    for sp, idx in task["other_indices"].items():
        eval_species[sp] = _get_worker_model(sp, idx)

    map_seed = task["map_seed"]
    python_random_seed = task["python_random_seed"]
    start_time = time.time()
    fitness, episode_length, end_reason = runner.run(
        eval_species,
        task["species"],
        map_seed,
        is_evaluation=False,
        callback=noop,
        collect_plot_data=False,
        python_random_seed=python_random_seed,
    )
    while episode_length == 0:
        if runner.settings.enable_logging:
            print("something went wrong update this seed")
        map_seed = random.randrange(100000)
        python_random_seed = random.randrange(2**32)
        fitness, episode_length, end_reason = runner.run(
            eval_species,
            task["species"],
            map_seed,
            is_evaluation=False,
            callback=noop,
            collect_plot_data=False,
            python_random_seed=python_random_seed,
        )

    end_time = time.time()
    return {
        "agent_index": task["agent_index"],
        "species": task["species"],
        "eval_index": task["eval_index"],
        "fitness": fitness,
        "episode_length": episode_length,
        "end_reason": end_reason,
        "steps_per_sec": episode_length / max(end_time - start_time, 1e-9),
    }
